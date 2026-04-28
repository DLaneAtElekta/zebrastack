Configuration IISStaticSite {
    param (
        [string]$WebsiteName = "ActivationAtlas",
        [string]$PhysicalPath = "C:\dev\activation-atlas",
        [string]$Port = 80
    )

    Import-DscResource -ModuleName PSDesiredStateConfiguration
    Import-DscResource -ModuleName xWebAdministration

    Node localhost {
        # Ensure IIS is installed
        WindowsFeature IIS {
            Ensure = "Present"
            Name   = "Web-Server"
        }

        WindowsFeature IISManagementTools {
            Ensure    = "Present"
            Name      = "Web-Mgmt-Tools"
            DependsOn = "[WindowsFeature]IIS"
        }

        # Remove default website if it exists
        xWebsite DefaultSite {
            Ensure       = "Absent"
            Name         = "Default Web Site"
            PhysicalPath = "C:\inetpub\wwwroot"
            DependsOn    = "[WindowsFeature]IIS"
        }

        # Create the static website
        xWebsite StaticWebsite {
            Ensure       = "Present"
            Name         = $WebsiteName
            State        = "Started"
            PhysicalPath = $PhysicalPath
            BindingInfo  = @(
                MSFT_xWebBindingInformation {
                    Protocol              = "HTTP"
                    Port                  = $Port
                    IPAddress             = "*"
                }
            )
            DependsOn    = "[xWebsite]DefaultSite"
        }
    }
}

# Generate the MOF file
IISStaticSite -OutputPath "C:\dev\activation-atlas\IISStaticSite"

# Apply the configuration
# Start-DscConfiguration -Path "C:\dev\activation-atlas\IISStaticSite" -Wait -Verbose -Force
